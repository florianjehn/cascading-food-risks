import pickle

import networkx as nx
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

import plotly.graph_objects as go


with open('../data/name_to_iso_alpha.pickle', 'rb') as handle:
    name_to_iso_alpha = pickle.load(handle)
    name_to_iso_alpha['Pitcairn Islands'] = 'PCN'
    name_to_iso_alpha['Wallis and Futuna Islands'] = 'WLF'
    name_to_iso_alpha['Netherlands Antilles (former)'] = 'ANT'
    name_to_iso_alpha['French Southern and Antarctic Territories'] = 'ATF'


with open('../data/name_to_centroid.pickle', 'rb') as handle:
    name_to_centroid = pickle.load(handle)
    name_to_centroid['Congo (Dem. Rep.)'] = (23.64396107, -2.87746289)
    name_to_centroid['Democratic Republic of the Congo'] = (23.64396107, -2.87746289)
    name_to_centroid['Cote d’Ivoire'] = (-5.5692157, 7.6284262)
    name_to_centroid['Laos'] = (102.4954987, 19.8562698)
    name_to_centroid['South Korea'] = (127.83916086, 36.38523983)
    name_to_centroid['Czech Republic'] = (15.31240163, 49.73341233)


def get_iso_alpha_from_country_name(name: str):
    return name_to_iso_alpha[name]


def get_lon_lat_from_country_name(name: str):
    return name_to_centroid[name]


class TradeGraph:

    def __init__(self, trade_data: pd.DataFrame,
                 locust_data: pd.DataFrame, poverty_data: pd.DataFrame):
        """
        Import / exports
        Locust vulnerability
        Population underneath the poverty line

        trade_data must have columns 'partner', 'reporter', 'element', 'value'
        country_data must have columns
        """

        self.G = nx.DiGraph()
        self.locust_data = locust_data
        self.poverty_data = poverty_data

        # Add node data (countries)
        for country in trade_data.partner.unique():

            if country == 'Unspecified Area':
                continue

            self.G.add_node(
                country,
                name=country,
                iso=get_iso_alpha_from_country_name(country),
                centroid=get_lon_lat_from_country_name(country),
                production=0,
                population=0,
                deficit=0,
                deficit_relative=0,
                locust_start='',
                locust_end='',
                covid_start='',
                covid_end='',
            )

        # Add edge data (trade)
        for i, row in trade_data.iterrows():

            edge = self.create_nx_edge_from_trade_row(row)

            if 'Unspecified Area' in edge:
                continue

            if self.G.has_edge(*edge):
                self.G.add_edge(
                    *edge,
                    amount=self.get_combined_trade_estimate(*edge, row.value)
                )
            else:
                self.G.add_edge(*edge, amount=row.value)

        self.sum_of_trade = sum(d['amount'] for _, _, d in self.G.edges(data=True))

    def get_combined_trade_estimate(self, u: str, v: str, new_amount: float):
        """
        It's possible that two countries can have conflicting reports
        for the same quantity, e.g. Australia says New Zealand exports
        100 units of food to Australia, but New Zealand says this figure
        is 110. This function resolves any conflicts by taking the mean.
        """
        return (self.G.get_edge_data(u, v)['amount'] + new_amount) / 2

    def plot_overall_trade_graph(self):
        """
        Using networkx draw method, plot all trade represented in the
        trade graph. Note this this currently only works in a Jupyter notebook
        """

        # Make edge size dependent on amount - i.e. volume of trade
        amounts = [self.G[u][v]['amount'] for u, v in self.G.edges()]
        max_amount = max(amounts)
        weights = [max((50 * float(i) / max_amount), 0.005) for i in amounts]

        # Project graph onto a map using networkx draw methods
        crs = ccrs.PlateCarree()
        fig, ax = plt.subplots(
            1, 1, figsize=(72, 32),
            subplot_kw=dict(projection=crs))
        ax.coastlines()
        nx.draw_networkx(
            self.G,
            ax=ax,
            font_size=16,
            alpha=.4,
            width=weights,
            node_size=0,
            with_labels=False,
            pos=list(nx.get_node_attributes(self.G, 'centroid').values()),
            edge_cmap=plt.cm.winter,
        )

    def get_export_dict(self, country):
        return {e[1]: self.G.get_edge_data(country, e[1])['amount'] for e
                in self.G.out_edges(country)}

    def get_import_dict(self, country):
        return {e[0]: self.G.get_edge_data(e[0], country)['amount'] for e
                in self.G.in_edges(country)}

    def get_import_sum(self, country):
        return sum(self.G.get_edge_data(e[0], country)['amount'] for e
                in self.G.in_edges(country))

    def get_export_sum(self, country):
        return sum(self.G.get_edge_data(country, e[1])['amount'] for e
                in self.G.out_edges(country))

    def reset_deficits(self):
        nx.set_node_attributes(self.G, 0, 'deficit')
        nx.set_node_attributes(self.G, 0, 'deficit_relative')

    def get_absolute_deficit(self, source, partner, reduction):
        return self.G.get_edge_data(source, partner)['amount'] * reduction

    def get_relative_deficit(self, source, partner, reduction):
        return (self.get_absolute_deficit(source, partner, reduction) /
                self.get_import_sum(partner)) * 100

    def apply_deficits(self, source, reduction):
        nx.set_node_attributes(
            self.G,
            {partner: nx.get_node_attributes(self.G, 'deficit')[partner] +
                      self.get_absolute_deficit(source, partner, reduction)
             for _, partner in self.G.out_edges(source)},
            'deficit'
        )

    def update_relative_deficits(self):
        for country, data in self.G.nodes(data=True):
            nx.set_node_attributes(
                self.G,
                {country: (data['deficit'] / self.get_import_sum(country)) * 100},
                'deficit_relative'
            )

    def list_node_attributes(self, attr):
        return list(nx.get_node_attributes(self.G, attr).values())

    def plot_export_restriction_scenario(self, scenario: dict, title=None,
                                         show_locusts=False, show_poverty=False,
                                         map_type=None):
        """
        Expect scenario to take form:

        {
            "Russian Federation": 0.9,  # exports reduced by 10%
            "India": 0.7,  # exports reduced by 30%
            ...
        }
        """

        # Calculate deficits

        self.reset_deficits()

        for source, export_fraction in scenario.items():
            if source not in self.G.nodes:
                raise ValueError(f'{source} is not in graph')
            self.apply_deficits(source, 1-export_fraction)

        total_deficits = sum(self.list_node_attributes('deficit'))
        total_relative_deficit = (total_deficits / self.sum_of_trade) * 100

        self.update_relative_deficits()

        # Plot Chloropleth map with deficits
        fig = go.Figure(
            data=go.Choropleth(
                locations=self.list_node_attributes('iso'),
                z=self.list_node_attributes('deficit_relative'),
                text=self.list_node_attributes('name'),
                colorscale='Reds',
                autocolorscale=False,
                marker_line_color='darkgray',
                marker_line_width=0.5,
                colorbar_title='Food Import Deficit (%)',
                colorbar_len=0.5,
                colorbar_title_font_size=14
            )
        )
        if map_type:
            fig.update_geos(projection_type=map_type)

        # Add lines to the map showing trade edges
        def get_trade_line_width(u, v):
            amount = self.G[u][v]['amount']
            return max(amount / 1e6, 0.01)

        for source, partner in self.G.edges():
            if source in scenario:
                fig.add_trace(
                    go.Scattergeo(
                        lon=[get_lon_lat_from_country_name(source)[0],
                             get_lon_lat_from_country_name(partner)[0]],
                        lat=[get_lon_lat_from_country_name(source)[1],
                             get_lon_lat_from_country_name(partner)[1]],
                        mode='lines',
                        line=dict(width=get_trade_line_width(source,  partner),
                                  color='#2E9246'),
                        opacity=0.5,
                        showlegend=False,
                    )
                )

        # Add poverty data
        if show_poverty:
            fig.add_trace(
                go.Scattergeo(
                    lon=self.poverty_data.country.apply(lambda c: get_lon_lat_from_country_name(c)[0]),
                    lat=self.poverty_data.country.apply(lambda c: get_lon_lat_from_country_name(c)[1]),
                    opacity=0.9,
                    showlegend=False,
                    marker=dict(
                        size=self.poverty_data.percent_poverty.apply(lambda p: p / 3),
                        opacity=1,
                        color='royalblue'
                    )
                )
            )

        # Add locust data
        if show_locusts:
            risk_to_color = {
                'High': 'red',
                'Medium': 'orange',
                'Low': 'yellow'
            }
            fig.add_trace(
                go.Scattergeo(
                    lon=self.locust_data.country.apply(
                        lambda c: get_lon_lat_from_country_name(c)[0]),
                    lat=self.locust_data.country.apply(
                        lambda c: get_lon_lat_from_country_name(c)[1]),
                    showlegend=False,
                    mode='markers',
                    marker=dict(
                        symbol='square',
                        opacity=1,
                        color=self.locust_data.locust_risk.apply(lambda r: risk_to_color[r]),
                    )
                )
            )

        # Apply figure styling
        fig.update_layout(
            title_font_size=18,
            title={
                'text': title or f"Effects of {', '.join(scenario.keys())} "
                        f"restricting exports",
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            annotations=[dict(
                x=0.55,
                y=0,
                xref='paper',
                yref='paper',
                text=f'{round(total_relative_deficit)}% '
                        f'global trade shortfall',
                showarrow=False
            )],
            width=1500,
            height=1000,
            geo=dict(
                showland=True,
                landcolor='rgb(243, 243, 243)',
                countrycolor='rgb(204, 204, 204)',
            ),
        )

        fig.show()

    @staticmethod
    def create_nx_edge_from_trade_row(row):
        if row.element == 'Import Quantity':
            return row.partner, row.reporter
        elif row.element == 'Export Quantity':
            return row.reporter, row.partner
        else:
            print(f'Unknown edge type: {row.element}')

